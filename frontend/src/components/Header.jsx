import logo from '../assets/logo.png'
import style from './Header.module.css'

export default function Header() {
    return <header className={style.header}>
          <img width="208" height="73" src={logo} alt="Hunter Badminton Association logo"/>
        </header>
}